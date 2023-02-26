import os
import cv2 as cv
from PIL import Image
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import pickle
os.makedirs('tmp/processed', exist_ok=True)
os.makedirs('tmp/equalized', exist_ok=True)
os.makedirs('tmp/filtered', exist_ok=True)

global bwimg


def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def inBounds(coord, dim):
    return coord > dim/3 and coord < dim/3 * 2

# for filename in os.listdir('cancer/'):
#     # Load the image and convert to grayscale
#     img = Image.open('cancer/' + filename).convert('L')

#     # Scale the image to the min and max grayscale values
#     img_arr = np.array(img)

#     # find the min and max color sections
#     min_val = np.min(img_arr)
#     max_val = np.max(img_arr)
#     # scale the image grayscale values based on the min and max color sections
#     img_scaled_arr = (img_arr - min_val) / (max_val - min_val)
#     img_scaled_arr = (img_scaled_arr * 255).astype(np.uint8)

#     # Save the scaled image
#     img_scaled = Image.fromarray(img_scaled_arr)
#     img_scaled.save('tmp/processed/' + filename)

#     bwimg = cv.imread(cv.samples.findFile('tmp/processed/' + filename), 0)
#     bwimg_cropped = crop_img(bwimg, 0.8)
#     eq = cv.equalizeHist(bwimg_cropped)
#     dst = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
#     #dst = eq

#     lower_bound = 0
#     upper_bound = 20
#     mask = cv.threshold(dst, 127, 255, cv.THRESH_BINARY)
#     contours = cv.findContours(
#         mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     #blob = max(contours, key=lambda el: cv.contourArea(el))
#     #M = cv.moments(contours)
#     # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#     # cv.circle(dst, center, 2, (0, 0, 255), -1)
#     cv.drawContours(dst, contours, -1, (0, 255, 0), 3)

#     Image.fromarray(dst).save('tmp/equalized/' + filename)
# # cv.imshow('Source image', bwimg)
# # hist = cv.calcHist([dst], [0], None, [256], [0, 256])
# # plt.hist(hist)
# # plt.show()
# # cv.imshow('Equalized Image', dst)


# cv.waitKey()

global thres
def process_image(filename):
    # if file exists, skip
    if os.path.isfile('tmp/equalized/' + filename):
        return
    imgcolor = cv.imread('images/' + filename)
    kernel = np.ones((15, 15), np.uint8)

    shave = cv.morphologyEx(imgcolor, cv.MORPH_CLOSE, kernel, iterations=2)
    blur = cv.blur(shave, (15, 15))

    img = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    dst = crop_img(cv.equalizeHist(img), 0.75)
    (T, threshInv) = cv.threshold(dst, 40, 255, cv.THRESH_BINARY_INV)

    contours, hierarchy = cv.findContours(
        threshInv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return
    blob = max(contours, key=lambda el: cv.contourArea(el))
    #blob = sorted(contours, key=lambda el: cv.contourArea(el))
    M = cv.moments(blob)
    if M["m00"] == 0:
        return
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # print(center)
    (x, y, w, h) = cv.boundingRect(blob)
    dim = dst.shape
    if x - 200 > 0:
        x = x - 200
    else:
        x = 0
    if y - 200 > 0:
        y = y-200
    else:
        y = 0
    if w + 200 < dim[0]:
        w = w+200
    else:
        w = dim[0]
    if h + 200 < dim[1]:
        h = h+200
    else:
        h = dim[1]

    mask = np.zeros_like(dst)
    cv.drawContours(mask, contours, -1, (255, 255, 255), cv.FILLED)
    cv.drawContours(mask, blob, -1, (255, 255, 255), 400)
    result = cv.bitwise_and(dst, mask)
    blackMask = cv.inRange(result, 0, 10)
    percent_black = cv.countNonZero(blackMask)/result.size
    include = False
    if (percent_black * 100) < 97 and inBounds(center[0], img.shape[0]):
        include = True
    #stencil = np.zeros(dst.shape).astype(dst.dtype)
    #cv.fillPoly(stencil, blob, [255, 255, 255])
    #result = cv.bitwise_and(dst, stencil)
    #cv.drawContours(dst, scale_contour(blob, 2), -1, (255, 255, 255), 20)
    #cv.circle(dst, center, 10, (255, 255, 255), 10)
    Image.fromarray(result).save('tmp/equalized/' + filename)
    if include:
        Image.fromarray(result).save('tmp/filtered/' + filename)

if __name__ == '__main__':
    # read dataset from pickle file
    with open('processed_dataset.pickle' , 'rb') as f:
        dataset = pickle.load(f)
    pool = Pool(processes=mp.cpu_count()-2)
    with tqdm(total=len(os.listdir('images/'))) as pbar:
        for _ in pool.imap_unordered(process_image, os.listdir('images/')):
            pbar.update()
    pool.close()
    pool.join()
