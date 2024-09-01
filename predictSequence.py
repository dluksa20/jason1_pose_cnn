import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

# Load your trained model
# model = load_model('Trained_models/darknet19_model_best_checkpoint.h5', compile=False)
model = load_model('Trained_models/darknet19.h5', compile=False)

directory = '/Users/dovydasluksa/Documents/Project_MSc/Test_images_2/'

# Sort filenames
sorted_filenames = sorted(os.listdir(directory))

for filename in sorted_filenames:
    if filename.endswith('.jpg') or filename.endswith('.png'): # check for image file extensions
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=(640, 640))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0

        prediction = model.predict(x)

        print(prediction, filename)
