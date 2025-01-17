import numpy as np
import pandas as pd
from data_preprocessing import prepare_for_train_test
from model_architecture import create_model
from train_eval import train_model, test_model, plot_model_training_curve
from glob import glob
import os
from PIL import Image

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join("Skin Cancer MNIST HAM10000/", '*', '*.jpg'))}
lesion_type_dict = {
    'nv': 'Melanocytic nevi (nv)',
    'mel': 'Melanoma (mel)',
    'bkl': 'Benign keratosis-like lesions (bkl)',
    'bcc': 'Basal cell carcinoma (bcc)',
    'akiec': 'Actinic keratoses (akiec)',
    'vasc': 'Vascular lesions (vasc)',
    'df': 'Dermatofibroma (df)'
}
label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}
reverse_label_mapping = dict((value, key) for key, value in label_mapping.items())

data = pd.read_csv(os.path.join("Skin Cancer MNIST HAM10000/",'HAM10000_metadata.csv'))

data['age'].fillna(value=int(data['age'].mean()), inplace=True)
data['age'] = data['age'].astype('int32')

data['cell_type'] = data['dx'].map(lesion_type_dict.get)
data['path'] = data['image_id'].map(imageid_path_dict.get)

data['image_pixel'] = data['path'].map(lambda x: np.asarray(Image.open(x).resize((28,28))))

data['label'] = data['dx'].map(reverse_label_mapping.get)

data = data.sort_values('label')
data = data.reset_index()

counter = 0
frames = [data]
for i in [4,4,11,17,45,52]:
    counter+=1
    index = data[data['label'] == counter].index.values
    df_index = data.iloc[int(min(index)):int(max(index)+1)]
    df_index = df_index.append([df_index]*i, ignore_index = True)
    frames.append(df_index)
    
final_data = pd.concat(frames)

X_orig = data['image_pixel'].to_numpy()
X_orig = np.stack(X_orig, axis=0)
Y_orig = np.array(data.iloc[:, -1:])
print(X_orig.shape)
print(Y_orig.shape)

X_aug = final_data['image_pixel'].to_numpy()
X_aug = np.stack(X_aug, axis=0)
Y_aug = np.array(final_data.iloc[:, -1:])
print(X_aug.shape)
print(Y_aug.shape)

X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = prepare_for_train_test(X_orig, Y_orig)

model =create_model()

X_train_aug, X_test_aug, Y_train_aug, Y_test_aug = prepare_for_train_test(X_aug, Y_aug)

model2_history = train_model(model, X_train_aug, Y_train_aug, 20)

plot_model_training_curve(model2_history)

test_model(model, X_train_aug, Y_train_aug, X_test_orig, Y_test_orig, label_mapping, data)

##########################################################################################
image_path ='/home/mostafatarek/Desktop/000my/study/000/Skin/HAM10000/Skin Cancer MNIST HAM10000/HAM10000_images_part_1/ISIC_0029297.jpg' 
img = np.asarray(Image.open(image_path).resize((28,28)))
new_one = img.reshape((1,28,28,3))
v = model.predict(new_one)
max_index = np.argmax(v)

print("Index of max value:", max_index)