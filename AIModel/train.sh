#!/bin/bash

# Download URLs of the zip files
url1="https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
url2="https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/download?datasetVersionNumber=2"

# Download the zip files
curl -OL "$url1" 
curl -OL "$url2"

# Unzip the downloaded zip files
unzip ISIC_2020_Training_JPEG.zip
unzip archive.zip

mv HAM10000_images_part_1 images
mv HAM10000_images_part_2 images
mv train images

py prep.py
py process_datasets.py
py model.py

