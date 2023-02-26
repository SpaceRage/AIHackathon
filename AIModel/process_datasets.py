# %%
import pickle
import pandas as pd
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm

# %%
# find matching ids from two csv files
df1 = pd.read_csv('ISIC_2020_Training_GroundTruth_v2.csv')
df2 = pd.read_csv('HAM10000_metadata.csv')
df2 = df2.drop(['lesion_id', 'dx_type', ], axis=1)

df1 = df1.rename(columns={'image_name': 'image_id'})
df1 = df1.rename(columns={'diagnosis': 'dx'})
df1 = df1.rename(columns={'anatom_site_general_challenge': 'localization'})
df1 = df1.rename(columns={'age_approx': 'age'})
df1 = df1.drop(['benign_malignant', "patient_id", "lesion_id", "target"], axis=1)

df = pd.concat([df1, df2], axis=0, sort=False)

# %%
df['localization'] = df['localization'].replace('trunk', 'torso')
df['localization'] = df['localization'].replace('face', 'head/neck')
df['localization'] = df['localization'].replace('ear', 'head/neck')
df['localization'] = df['localization'].replace('scalp', 'head/neck')
df['localization'] = df['localization'].replace('abdomen', 'torso')
df['localization'] = df['localization'].replace('genital', 'oral/genital')
df['localization'] = df['localization'].replace('back', 'torso')
df['localization'] = df['localization'].replace('chest', 'torso')
df['localization'] = df['localization'].replace('neck', 'head/neck')
df['localization'] = df['localization'].replace('hand', 'hands/feet')
df['localization'] = df['localization'].replace('foot', 'hands/feet')
df['localization'] = df['localization'].replace('acral', 'hands/feet')
df.dropna(subset=['localization'], inplace=True)

df['dx'] = df['dx'].replace('nv', 'nevus')
df['dx'] = df['dx'].replace('cafe-au-lait macule', 'nevus')
df['dx'] = df['dx'].replace('atypical melanocytic proliferation', 'melanoma')
df['dx'] = df['dx'].replace('seborrheic keratosis', 'keratosis-like')
df['dx'] = df['dx'].replace('lichenoid keratosis', 'keratosis-like')
df['dx'] = df['dx'].replace('akiec', 'keratosis-like')
df['dx'] = df['dx'].replace('bkl', 'keratosis-like')
df['dx'] = df['dx'].replace('solar lentigo', 'lentigo')
df['dx'] = df['dx'].replace('lentigo NOS', 'lentigo')
df['dx'] = df['dx'].replace('mel', 'melanoma')
df['dx'] = df['dx'].replace('bcc', 'basal cell carcinoma')
df['dx'] = df['dx'].replace('vasc', 'vascular lesion')
df['dx'] = df['dx'].replace('df', 'dermatofibroma')
df = df[df.dx != 'unknown']

df['sex'] = df['sex'].replace('male', '1')
df['sex'] = df['sex'].replace('female', '1')
df = df[df['sex'] != 'unknown']


df.dropna(subset=['age'], inplace=True)
df['age'] = df['age'].astype(int)
df = df.set_index('image_id')
df

# %%
# list all unique values in localization as a list
localizations = df['localization'].unique()
localizations = localizations.tolist()
print(localizations)

# %%
# count of rows for each dx
diag = df['dx'].unique()
diag = diag.tolist()
print(diag)

# %%
df = df.sample(n=100, random_state=1)

# %%
# copy thoose images to a new folder
# make a new folder

os.makedirs('data', exist_ok=True)
for idx in tqdm(df.index):
    # check if image exists
    if os.path.exists(os.path.join('tmp/filtered/', idx + '.jpg')):
        img = Image.open(os.path.join('tmp/filtered', idx + '.jpg'))
        img.save(os.path.join('data', idx + '.jpg'))
    else:
        # drop the row
        df.drop(idx, inplace=True)
    

# %%
# create a new column for image
df['image'] = ''

# %%
for idx in tqdm(df.index):
    img = Image.open(os.path.join('tmp/equalized', idx + '.jpg'))
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)  # convert to NumPy array with float32 dtype
    df.at[idx, 'image'] = img

# %%
# save df, localizations, diag into pickle
with open('processed_dataset.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(localizations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(diag, handle, protocol=pickle.HIGHEST_PROTOCOL)


