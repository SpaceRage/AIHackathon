import os
import random
import shutil
from PIL import Image
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

os.makedirs('tmp/norm_gs', exist_ok=True)
os.makedirs('tmp/processed', exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)
os.makedirs('data/test', exist_ok=True)


def scale_grayscale_image(filename):
    # Load the image and convert to grayscale
    img = Image.open('tmp/norm_gs/' + filename).convert('L')

    # Scale the image to the min and max grayscale values
    img_arr = np.array(img)


    # find the min and max color sections
    min_val = np.min(img_arr)
    max_val = np.max(img_arr)
    # scale the image grayscale values based on the min and max color sections
    img_scaled_arr = (img_arr - min_val) / (max_val - min_val)
    img_scaled_arr = (img_scaled_arr * 255).astype(np.uint8)


    # Save the scaled image
    img_scaled = Image.fromarray(img_scaled_arr)
    img_scaled.save('tmp/processed/' + filename)


def process_image(filename):
    # Load the image and resize to 500x500
    img = Image.open('images/' + filename)
    img = img.resize((500, 500))

    # Convert image to grayscale
    img = img.convert('L')

    # Save the normalized grayscale image
    img.save('tmp/norm_gs/' + filename)

    # Scale the grayscale image
    scale_grayscale_image(filename)


def process_images():
    pool = mp.Pool()
    filenames = os.listdir('images')
    for _ in tqdm(pool.imap_unordered(process_image, filenames), total=len(filenames)):
        pass
    pool.close()
    pool.join()


def split_data():
    # Define source and destination folders
    source_folder = 'tmp/processed'
    train_folder = 'data/train'
    val_folder = 'data/val'
    test_folder = 'data/test'

    # Define the train/validation/test split ratio
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Get a list of all the files in the source folder
    file_list = os.listdir(source_folder)

    # Shuffle the file list
    random.shuffle(file_list)

    # Split the files into train, validation, and test sets
    train_files = file_list[:int(len(file_list) * train_ratio)]
    val_files = file_list[int(len(file_list) * train_ratio):int(len(file_list) * (train_ratio + val_ratio))]
    test_files = file_list[int(len(file_list) * (train_ratio + val_ratio)):]

    # Copy the train files to the train folder
    print('Copying train files...')
    for file in tqdm(train_files):
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

    # Copy the validation files to the validation folder
    print('Copying validation files...')
    for file in tqdm(val_files):
        shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))

    # Copy the test files to the test folder
    print('Copying test files...')
    for file in tqdm(test_files):
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))


if __name__ == '__main__':
    process_images()
    split_data()
    shutil.rmtree('tmp')
