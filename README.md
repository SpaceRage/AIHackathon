# SpotCheck - A submission to the Responsible AI Hackathon
## Team: SpotCheck
### Team Members:
- Adam
- Madhav
- Luisa
- Rajiv

## Problem Statement
To rectify the discrepancy in AI models underdiagnosing darker-skinned patients with melanoma due to being trained to better recognize cancer on light skin, our goal was to create a model trained on an image dataset which was neutral to skin tone and color to increase diagnostic accuracy.
## How to run the code
### Training
1. Download both datasets from:
    - https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
    - https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Unzip both datasets into `AIModel/images` in the AIModel folder. Extract the images from the ISIC_2020_Training_JPEG.zip into 'AIModel/images' and the images from the HAM10000_images_part_1 and the HAM10000_images_part_2 folders into 'AIModel/images'.
3. Create a new virtual environment  called SpotCheck with `python3 -m venv SpotCheck`
4. Install the requirements with `pip install -r requirements.txt`
5. cd into the AIModel folder
6. Run the training script with `train.sh`
7. The model will be saved in the `AIModel` folder as `skin_cancer_diagnosis_model.h5`

### Running the web app
1. Download https://drive.google.com/drive/folders/1eI2kVQ4_SLDMZpcSdTkHETtxRu4V329X?usp=sharing
1. Create a new virtual environment called SpotCheck with `python3 -m venv SpotCheck` and install the requirements with `pip install -r requirements.txt`
2. cd into the `AIBackend` folder
3. Run the web app with `python backend.py`
4. Open the web app at `localhost:5000`

#### Note: The app will crash if the computer does not have a GPU with enough VRAM to run the model. If this is the case, please run the app on a computer with a GPU with at least 4GB of VRAM.