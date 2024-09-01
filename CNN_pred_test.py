import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
from utils import PoseLosses, TrainableSPSQLayer, sixDimToRotMat
from keras.utils import custom_object_scope

loaded_model = load_model('Trained_models/darknet19.h5', compile=False)
# with custom_object_scope({'TrainableSPSQLayer': TrainableSPSQLayer, 'lossTotal': PoseLosses.Adaptive_loss}):
#     loaded_model = load_model('best_model.h5')

# Load the test dataset from a CSV file
test_data = pd.read_csv('dataset.csv')  # Adjust this path to your test dataset's location if necessary
# Uncomment the following line if you want to limit the test data to the first 200 entries
test_data = test_data.head(200)

# Preprocess the test data:
# 1. Extract the image paths from the CSV
# 2. Load each image using TensorFlow Keras utility, resizing them to 400x400
# 3. Convert the images to arrays and normalize pixel values to [0, 1]
image_paths = test_data['Image path']
test_images = [tf.keras.utils.load_img(path, target_size=(640, 640)) for path in image_paths]
test_images = np.array([tf.keras.utils.img_to_array(img)/255.0 for img in test_images])

# Use the trained model to make predictions on the test images
predictions = loaded_model.predict(test_images)

# Extract translation and rotation predictions from the model's output
translations = predictions[:, :3]  # First 3 values are translations ZYX
rotations = predictions[:, 3:]     # The remaining values are rotations

# Display the predicted results for each image
for i, (translation, rotation, image_path) in enumerate(zip(translations, rotations, image_paths)):
    print(f"Image {i+1}:")
    print(f"Image Path: {image_path}")
    print(f"Predicted Translation: {translation}")
    # Convert the 6D rotation vector to a rotation matrix using a utility function
    print(F"{sixDimToRotMat(rotation)}")
    print(f"Predicted Rotation: {rotation}")
    print("-----")
