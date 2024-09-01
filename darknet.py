import numpy as np
from typing import List, Tuple
from keras.models import Model, Sequential
from keras.layers import Input, Add, Conv2D, ZeroPadding2D, GlobalAveragePooling2D, Dense, Flatten, LeakyReLU, BatchNormalization, Concatenate
from keras.layers import LSTM, Reshape

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
    fc1 = Dense(3, name='fc1')(globalAvgPool)
    fc2 = Dense(6, name='fc2')(globalAvgPool)

    out = Concatenate(name='fc')([fc1, fc2])

    model = Model(inputs = x_input, outputs = out, name='darknet19')
    
    return model


if __name__ == '__main__':
    model = darknet19((480,744,3))
    model.summary()


