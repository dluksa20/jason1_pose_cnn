import tensorflow as tf
from keras.models import load_model
import numpy as np
from utils import sixDimToRotMat, so3_to_su2
import cv2

# Setting image dimensions for preprocessing
img_width, img_height = 640, 640

# Load a pre-trained Keras model from the specified path
model = load_model('Trained_models/darknet19.h5', compile=False)
# Read an image for testing using OpenCV
test_image = cv2.imread('/Users/dovydasluksa/Documents/Project_MSc/Test_images_2/frame_200.png')

test_image = cv2.resize(test_image, (img_width, img_height))# Resize the image to the predefined dimensions
# Expand the image dimensions, effectively adding a batch dimension to the image 
test_image = np.expand_dims(test_image, axis=0)
test_image = (test_image[...,::-1].astype(np.float32)) / 255.0# Normalize the image and also change its channel order from BGR to RGB (as OpenCV reads in BGR format)


prediction = model.predict(test_image)
print(prediction[0].tolist())


# Convert the 6D representation of rotation in the prediction to a 3x3 rotation matrix
rotMatPred = sixDimToRotMat(prediction[0][3:])
print(rotMatPred)

# Convert the 3x3 rotation matrix to a quaternion representation
# quatPred = so3_to_su2(rotMatPred)

# # Combine the translation (first 3 values) with the quaternion rotation for the final pose
# outputPose = np.concatenate((prediction[0][:3], quatPred))
# print(outputPose)
