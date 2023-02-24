import os
import pydicom
import pickle
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

os.makedirs('png', exist_ok=True)


def convert_to_png(filename):
    ds = pydicom.dcmread('train/' + filename)
    ds.save_as('png/' + filename + '.png')


def process_images():
    pool = mp.Pool()
    filenames = os.listdir('train')
    for _ in tqdm(pool.imap_unordered(convert_to_png, filenames), total=len(filenames)):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    process_images()

    # load the ground truth from the csv file
    df = pd.read_csv('ISIC_2020_Training_GroundTruth_v2.csv')
    # save the ground truth in a pickle file
    with open('ground_truth.pkl', 'wb') as f:
        pickle.dump(df, f)
