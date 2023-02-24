import os
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

os.makedirs('norm_gs', exist_ok=True)
os.makedirs('custom_gs', exist_ok=True)


def process_image(filename):
    # Convert image to grayscale and resize to 500x500
    img = cv2.imread('png/' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (500, 500))
    cv2.imwrite('norm_gs/' + filename, img)

    # Scale image to min and max grayscale values
    img = cv2.imread('norm_gs/' + filename, cv2.IMREAD_GRAYSCALE)
    img_min = np.amin(img)
    img_max = np.amax(img)
    img_scaled = np.interp(img, (img_min, img_max), (0, 255)).astype(np.uint8)
    cv2.imwrite('custom_gs/' + filename, img_scaled)


def process_images():
    pool = mp.Pool()
    filenames = os.listdir('png')
    for _ in tqdm(pool.imap_unordered(process_image, filenames), total=len(filenames)):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    process_images()
