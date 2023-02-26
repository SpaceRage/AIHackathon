# SpotCheck - A submission to the Responsible AI Hackathon
## Team: SpotCheck
### Team Members:
- Adam
- Madhav
- Luisa
- Rajiv

## Problem Statement

## How to run the code
### Training
1. Download both datasets from:
    - https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
    - https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Unzip both datasets into `AIModel/images` in the AIModel folder. Extract the images from the ISIC_2020_Training_JPEG.zip into 'AIModel/images' and the images from the HAM10000_images_part_1 and the HAM10000_images_part_2 folders into 'AIModel/images'.
3. cd into the AIModel folder
4. Create a new virtual environment  called SpotCheck with `python3 -m venv SpotCheck`
5. Install the requirements with `pip install -r requirements.txt`
6. Run the training script with `train.sh`
7. The model will be saved in the `AIModel` folder as `skin_cancer_diagnosis_model.h5`

### Running the web app
1. cd into the `AIBackend` folder