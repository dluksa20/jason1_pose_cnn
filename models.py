import numpy as np
from typing import List, Tuple
from keras.models import Model, Sequential
from keras.layers import Input, Add, Conv2D, ZeroPadding2D, GlobalAveragePooling2D, Dense, Flatten, LeakyReLU, BatchNormalization, Concatenate
from keras.layers import Input, DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Reshape, Dense, Multiply, Add, Dropout, Flatten, Concatenate
from keras.regularizers import l1, l2
from utils import TrainableSPSQLayer
import math
from keras.applications import EfficientNetB0

# ---------------------------------------------------------------------------------------------------------------------#
#                                             EfficientNetB0 base model                                                #
# ---------------------------------------------------------------------------------------------------------------------#


def EfficientNet(input_size):
    """
    Construct a model based on EfficientNetB0 architecture for pose estimation.

    Args:
    input_size (tuple): Expected input shape for the model (height, width, channels).

    Returns:
    Model: A Keras model object.

    Description:
    This function initializes a base EfficientNetB0 model without the top classification layers.
    The model is then extended to compute a 6-DOF (Degree of Freedom) pose, which consists of 
    a 3D translation (ZYX) and a 3D rotation (6DOF vector). Finally, a custom layer `CustomTrainableParamsLayer`(to
    train SP and SQ loss function weights) is applied to the concatenated outputs.
    """

    # Initialize the base model without the top (classification) layers.
    base_model = EfficientNetB0(include_top=False, input_shape=input_size, weights=None)
    pool_layer = GlobalAveragePooling2D(name='globalAvgPool')(base_model.output)
    
    # Compute a 3D translation output from the pooled features.
    translation_output = Dense(3, name='translation_output', kernel_regularizer=l1(0.01))(pool_layer)
    # Compute a 6D rotation output from the pooled features.
    rotation_output = Dense(6, name='rotation_output', kernel_regularizer=l1(0.01))(pool_layer)

    # Introduce loss function trainable weights SP and SQ as an addtional layer 
    concatenated_output = Concatenate(name='6DOF_pose')([translation_output, rotation_output])

    '''''
     for main_eager_execution.py execution 
     final_output - commented 
     for outputs passed concatenated_output

    '''''
    final_output = TrainableSPSQLayer()(concatenated_output)
    
    # Construct model
    model = Model(inputs=base_model.input, outputs=final_output, name='EfficientNetB0')

    return model
# if __name__ == '__main__':
#     model = EfficientNet((640,640,3))
#     print(len(model.layers))
#     model.summary()

# ---------------------------------------------------------------------------------------------------------------------#
#                                             EfficientNetB0 scaled model                                              #
# ---------------------------------------------------------------------------------------------------------------------#
"""
Scaling applied based on:

Reference:
Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
arXiv preprint arXiv:1905.11946.

First convolutional layer kernel size increased from 3x3 to 7x7

"""


# Squeeze and Excitation block
def SEBlock(inputs, ratio=4):
    channels = inputs.shape[-1]
    squeezed = GlobalAveragePooling2D()(inputs)
    squeezed = Reshape((1, 1, channels))(squeezed)
    excitation = Dense(channels // ratio, activation='relu')(squeezed)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    return Multiply()([inputs, excitation])

# MBConv block
def MBConv(inputs, in_channels, out_channels, expansion_factor, kernel_size, stride, se_ratio=4, drop_rate=0.2):
    x = Conv2D(in_channels * expansion_factor, (1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = SEBlock(x, ratio=se_ratio)
    
    x = Conv2D(out_channels, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    if in_channels == out_channels and stride == 1:
        x = Add()([inputs, x])
    if drop_rate:
        x = Dropout(drop_rate)(x)
    
    return x

# Calculated multipliers
'''''
input shape 640x640x3
# d = 1.2 ** 7.5
# w = 1.1 ** 7.5

'''''
'''''
input shape 320x320x3
# d = 1.2 ** 3.75
# w = 1.1 ** 3.75

'''''
d = 1.2 ** 3.75
w = 1.1 ** 3.75
# Round a number of channels based on width multiplier
def round_channels(channels, multiplier):
    return int(math.ceil(channels * multiplier))

def build_efficientnet_b0_scaled(input_shape=(640, 640, 3)):
    inputs = Input(shape=input_shape)
    
    # Initial conv layer
    x = Conv2D(round_channels(32, w), (3, 3), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # MBConv blocks
    num_repeats = lambda rep: int(math.ceil(rep * d))
    
    x = MBConv(x, round_channels(32, w), round_channels(16, w), 1, (3, 3), 1)
    for _ in range(num_repeats(1)):
        x = MBConv(x, round_channels(16, w), round_channels(24, w), 6, (3, 3), 2)
    for _ in range(num_repeats(2)):
        x = MBConv(x, round_channels(24, w), round_channels(40, w), 6, (5, 5), 2)
    for _ in range(num_repeats(2)):
        x = MBConv(x, round_channels(40, w), round_channels(80, w), 6, (3, 3), 2)
    for _ in range(num_repeats(3)):
        x = MBConv(x, round_channels(80, w), round_channels(112, w), 6, (5, 5), 1)
    for _ in range(num_repeats(3)):
        x = MBConv(x, round_channels(112, w), round_channels(192, w), 6, (5, 5), 2)
    for _ in range(num_repeats(4)):
        x = MBConv(x, round_channels(192, w), round_channels(320, w), 6, (3, 3), 1)

    # Top layers
    x = Conv2D(round_channels(1280, w), (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    
    # Rotation and translation heads
    rotation = Dense(6, name="rotation", kernel_regularizer=l1(0.01))(x)  
    translation = Dense(3, name="translation", kernel_regularizer=l1(0.01))(x)  # 3D translation vector
    out_layer = Concatenate(name='6DOF_pose')([translation, rotation])

    '''''
     for main_eager_execution.py execution 
     final_output - commented 
     for outputs passed concatenated_output - 'out_layer'
     
    '''''

    final_output = TrainableSPSQLayer()(out_layer)

    model = Model(inputs=inputs, outputs=final_output, name='EfficientNetB0')    

    return model

# model = build_efficientnet_b0_scaled((640,640,3))
# print(len(model.layers))
# model.summary()

# ---------------------------------------------------------------------------------------------------------------------#
#                                                   darknet19 model                                                    #
# ---------------------------------------------------------------------------------------------------------------------#


def darknet19(input_size: Tuple[int,int,int]):
    #model   = Sequential()
    # Conv1
    x_input  = Input(input_size,name='input')
    padding = ZeroPadding2D((3, 3))(x_input)
    conv1   = Conv2D(32, kernel_size=7, strides=(2,2), padding='valid', use_bias=False, name='conv1')(padding)
    bn1     = BatchNormalization(name='bn1')(conv1)
    act1    = LeakyReLU(alpha=0.1, name='lrelu1')(bn1)

    # Conv2
    padding = ZeroPadding2D((1,1))(act1)
    conv2   = Conv2D(64, kernel_size=3, strides=(2,2), padding='valid', use_bias=False, name='conv2')(padding)
    bn2     = BatchNormalization(name='bn2')(conv2)
    act2    = LeakyReLU(alpha=0.1, name='lrelu2')(bn2)

    #Conv3
    padding = ZeroPadding2D((1,1))(act2)
    conv3_1 = Conv2D(128, kernel_size=3, strides=(1,1), padding='valid', use_bias=False, name='conv3_11')(padding)
    bn3_1   = BatchNormalization(name='bn3_11')(conv3_1)
    act3_1  = LeakyReLU(alpha=0.1, name='lrelu3_11')(bn3_1)
    conv3_2 = Conv2D(64,kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv3_12')(act3_1)
    bn3_2   = BatchNormalization(name='bn3_12')(conv3_2)
    act3_2  = LeakyReLU(alpha=0.1, name='lrelu3_12')(bn3_2)
    act3    = Add()([act2, act3_2])

    #Conv4
    padding = ZeroPadding2D((1,1))(act3)
    conv4   = Conv2D(128, kernel_size=3, strides=(2,2), padding='valid', use_bias=False, name='conv4')(padding)
    bn4     = BatchNormalization(name='bn4')(conv4)
    act4    = LeakyReLU(alpha=0.1, name='lrelu4')(bn4)

    #Conv5
    padding = ZeroPadding2D((1,1))(act4)
    conv5_1 = Conv2D(256, kernel_size=3, strides=(1,1), padding='valid', use_bias=False, name='conv5_11')(padding)
    bn5_1   = BatchNormalization(name='bn5_11')(conv5_1)
    act5_1  = LeakyReLU(alpha=0.1, name='lrelu5_11')(bn5_1)
    conv5_2 = Conv2D(128, kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv5_12')(act5_1)
    bn5_2   = BatchNormalization(name='bn5_12')(conv5_2)
    act5_2  = LeakyReLU(alpha=0.1, name='lrelu5_12')(bn5_2)
    act5    = Add()([act4, act5_2])

    #Conv6
    padding = ZeroPadding2D((1,1))(act5)
    conv6   = Conv2D(256, kernel_size=3, strides=(2,2), padding='valid', use_bias=False, name='conv6')(padding)
    bn6     = BatchNormalization(name='bn6')(conv6)
    act6    = LeakyReLU(alpha=0.1, name='lrelu6')(bn6)

    #Conv7
    padding = ZeroPadding2D((1,1))(act6)
    conv7_1 = Conv2D(512, kernel_size=3, strides=(1,1), padding='valid', use_bias=False, name='conv7_11')(padding)
    bn7_1   = BatchNormalization(name='bn7_11')(conv7_1)
    act7_1  = LeakyReLU(alpha=0.1, name='lrelu7_11')(bn7_1)
    conv7_2 = Conv2D(256, kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv7_12')(act7_1)
    bn7_2   = BatchNormalization(name='bn7_12')(conv7_2)
    act7_2  = LeakyReLU(alpha=0.1, name='lrelu7_12')(bn7_2)
    act7_1_1= Add()([act6, act7_2])

    padding = ZeroPadding2D((1,1))(act7_1_1)
    conv7_3 = Conv2D(512, kernel_size=3, strides=(1,1), padding='valid', use_bias=False, name='conv7_21')(padding)
    bn7_3   = BatchNormalization(name='bn7_21')(conv7_3)
    act7_3  = LeakyReLU(alpha=0.1, name='lrelu7_21')(bn7_3)
    conv7_4 = Conv2D(256, kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv7_22')(act7_3)
    bn7_4   = BatchNormalization(name='bn7_22')(conv7_4)
    act7_4  = LeakyReLU(alpha=0.1, name='lrelu7_22')(bn7_4)
    act7    = Add()([act7_1_1, act7_4])

    #Conv8
    padding = ZeroPadding2D((1,1))(act7)
    conv8   = Conv2D(512, kernel_size=3, strides=(2,2), padding='valid', use_bias=False, name='conv8')(padding)
    bn8     = BatchNormalization(name='bn8')(conv8)
    act8    = LeakyReLU(alpha=0.1, name='lrelu8')(bn8)

    #Conv9
    padding = ZeroPadding2D((1,1))(act8)
    conv9_1 = Conv2D(1024, kernel_size=3, strides=(1,1), padding='valid', use_bias=False, name='conv9_11')(padding)
    bn9_1   = BatchNormalization(name='bn9_11')(conv9_1)
    act9_1  = LeakyReLU(alpha=0.1, name='lrelu9_11')(bn9_1)
    conv9_2 = Conv2D(512, kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv9_12')(act9_1)
    bn9_2   = BatchNormalization(name='bn9_12')(conv9_2)
    act9_2  = LeakyReLU(alpha=0.1, name='lrelu9_12')(bn9_2)
    act9_1_1= Add()([act8, act9_2])
    padding = ZeroPadding2D((1,1))(act9_1_1)
    conv9_3 = Conv2D(1024, kernel_size=3, strides=(1,1), padding='valid', use_bias=False, name='conv9_21')(padding)
    bn9_3   = BatchNormalization(name='bn9_21')(conv9_3)
    act9_3  = LeakyReLU(alpha=0.1, name='lrelu9_21')(bn9_3)
    conv9_4 = Conv2D(512, kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv9_22')(act9_3)
    bn9_4   = BatchNormalization(name='bn9_22')(conv9_4)
    act9_4  = LeakyReLU(alpha=0.1, name='lrelu9_22')(bn9_4)
    act9    = Add()([act9_1_1, act9_4])

    #Conv10
    padding = ZeroPadding2D((1,1))(act9)
    conv10  = Conv2D(1024, kernel_size=3, strides=(1,1),padding='valid', use_bias=False, name='conv10')(padding)
    bn10    = BatchNormalization(name='bn10')(conv10)
    act10   = LeakyReLU(alpha=0.1, name='lrelu10')(bn10)

    #Conv11
    conv11 = Conv2D(1000, kernel_size=1, strides=(1,1), padding='valid', use_bias=False, name='conv11')(act10)

    globalAvgPool = GlobalAveragePooling2D(name='globalAvgPool')(conv11)

    fc1 = Dense(3, name='fc1', kernel_regularizer=l1(0.01))(globalAvgPool)
    fc2 = Dense(6, name='fc2', kernel_regularizer=l1(0.01))(globalAvgPool)

    out = Concatenate(name='fc')([fc1, fc2])

    '''''
     for main_eager_execution.py execution 
     final_output - commented 
     for outputs passed concatenated_output - 'out'
     
    '''''
    final_output = TrainableSPSQLayer()(out)

    model = Model(inputs = x_input, outputs = final_output, name='darknet19')
    
    return model


# if __name__ == '__main__':
#     model = darknet19((640,640,3))
#     print(len(model.layers))
#     model.summary()


