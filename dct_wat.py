
import numpy as np
import cv2 as cv
import random
import math


def dct_encode(img, wm, key, wm_size):
    '''
    Function embeds watermark into grayscale image. The watermark must be binary.
    Watermark can be automatically downscaled so it fits into image at least 8 times.

    Parameter
    ------------
    img : cv.Mat | 2d Array
    Matrix containing grayscale image

    wm : cv.Mat | 2d Array
    binary watermark that should be embedded

    key : int | Seed for random operations

    wm_size : [int,int] | Tuple representing watermark desired size.
    The same size must be given to decoding part
    '''

    # Resize to desired size
    wm = cv.resize(wm, wm_size)
    # Calculate number of watermarks that fits
    n_wm = wm_fits(img, 8, wm_size)
    # If not enough size for watermark insertion
    if n_wm < 8:
        print("Original image is to small to embed 8 watermarks into it")
        print("Change the watermark size or image size")
        return img
    # Embed the watermark
    watermark = bin_watermark(img, wm, key, n_wm)
    return watermark


def wm_fits(img, block_size, res_wm):
    '''

    :param img: cv.Mat | 2d Array
    :param block_size:
    :param res_wm:
    :return:
    '''
    res_img = img.shape
    blocks = int(res_img[0] / block_size) * int(res_img[1] / block_size)
    block_per_wm = int((res_wm[0] * res_wm[1]) / 8)

    return int(blocks / block_per_wm)


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
    for i in range(0, len(part_wm)):
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


def bin_watermark(orig, wm, key, num_wm):
    _block_size = 8
    # Output img
    embed_img = orig.copy()

    # Shuffle the watermark to mitigate spatial correlation
    rav_wm = wm.ravel()
    order = list(range(len(rav_wm)))
    random.seed(key)
    random.shuffle(order)
    rav_wm = [rav_wm[i] for i in order]

    # Randomly generate 8 numbers representing cyclical shift for every embedded watermark
    np.random.seed(key + 1)
    shifts = np.random.randint(1, 32, num_wm)
    print(shifts)
    # Counters to help with indexing
    counter = 0
    wm_id = 0
    # DO the first shift
    shifted_wm = rav_wm[shifts[wm_id]:] + rav_wm[:shifts[wm_id]]
    wm_id = wm_id + 1
    # Store several watermarks to fit the whole image
    for i in range(0, orig.shape[0], _block_size):

        for j in range(0, orig.shape[1], _block_size):

            # Reset the indexing and shift next watermark vector
            if counter >= len(rav_wm):
                # Break the loop max amount of wm was reached
                if wm_id >= num_wm:

                    break

                counter = 0
                # Shift
                shifted_wm = rav_wm[shifts[wm_id]:] + rav_wm[:shifts[wm_id]]

                # Increment index
                wm_id = wm_id + 1

            # Extract one block from image
            block = orig[i:i + _block_size, j:j + _block_size]
            # Do dct
            dct_block = cv.dct(np.float32(block))
            # Embed watermark into image
            wm_block = assign_to_block(dct_block, shifted_wm[counter:counter + 8])
            counter += 8
            # Inverse dct to recreate img with watermark
            idct_block = cv.idct(wm_block)
            # Normalize the values to prevent overflow
            idct_block = idct_block.clip(0, 255)
            # Store the block in appropriate position
            embed_img[i:i + _block_size, j:j + _block_size] = idct_block

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


def dct_decode(img, key):


    return bin_extract(img, key, [64,64])


def calc_size(imsh, num_wm, block_size):

    x_wm = int(imsh[0] / block_size)
    y_wm = int(imsh[1] / block_size)
    res = int(x_wm * y_wm)
    return res


def bin_extract(orig, key, wm_size):
    _block_size = 8
    # Watermark length in pixels
    _wm_len = wm_size[0] * wm_size[1]
    num_wm = wm_fits(orig, _block_size, wm_size)
    # Recreate the shuffle order for backward reshuffling
    order = list(range(_wm_len))
    random.seed(key)
    random.shuffle(order)

    # Generate shift indexes for reverse shifts
    np.random.seed(key + 1)
    shifts = np.random.randint(1, 32, num_wm)
    print(shifts)

    # Return values
    wms = []
    decoded = np.zeros(_wm_len)
    counter = 0
    wm_id = 0
    # Break flag
    break_flag = False
    for i in range(0, orig.shape[0], _block_size):
        if break_flag:
            break
        for j in range(0, orig.shape[1], _block_size):

            # Extract one block
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
            if counter >= _wm_len:
                counter = 0
                # Reverse shift
                decoded = decoded.astype(int).tolist()
                decoded = decoded[-shifts[wm_id]:] + decoded[:-shifts[wm_id]]
                wm_id = wm_id + 1

                # Reverse shuffle
                reordered = inverse_reordering(decoded, order)
                # For rounding error clip the values and then reshape to wm shape
                b = reordered.clip(0, 255).reshape(wm_size)
                # Append to all wms
                wms.append(np.uint8(b))
                if wm_id >= num_wm:
                    break_flag = True
                    break
                decoded = np.zeros(_wm_len)
    return wms
