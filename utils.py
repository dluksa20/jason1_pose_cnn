import tensorflow as tf
import keras.backend as K
import math
import numpy as np
from keras.layers import Layer
import tensorflow as tf
import os


# ---------------------------------------------------------------------------------------------------------------------#
#                                                 Function Definitions                                                 #
# ---------------------------------------------------------------------------------------------------------------------#

"""
6DOF continious representation based on 

Reference:
Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019).
On the Continuity of Rotation Representations in Neural Networks.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5745-5753).
"""


def sixDimToRotMat(sixDim):
    """
    6DOF continious vector to rotation matrix according Gram-Schmidt method for orthonormalizing 
    
    Convert a 6D vector to a 3x3 rotation matrix using Gram-Schmidt orthonormalization.
    
    Args:
    - sixDim (tf.Tensor): Input tensor representing the 6D vector.

    Returns:
    - tf.Tensor: 3x3 rotation matrix.
    """
    # Cast to float32
    sixDim = tf.cast(sixDim, dtype=tf.float32)
    
    # Reshape the 6D tensor for further operations
    transArr = K.reshape(sixDim, (-1, 2, 3))
    transArr = tf.transpose(transArr, perm=[0, 2, 1])
    
    # Extract orthogonal vectors from the 6D representation
    x_raw = transArr[:, :, 0]
    y_raw = transArr[:, :, 1]
    
    # Orthonormalize vectors
    x = K.l2_normalize(x_raw, axis=1)
    z = tf.linalg.cross(x, y_raw)
    z = K.l2_normalize(z, axis=1)
    y = tf.linalg.cross(z, x)

    # Construct the rotation matrix
    rotMat = tf.stack([x, y, z], axis=1)
    rotMat = tf.transpose(rotMat, perm=[0, 2, 1])
    
    return rotMat


# Rotation matrix to quternion domain
def so3_to_su2(mat):

    """
    Convert a 3x3 rotation matrix into a quaternion.
    
    Args:
    - mat (tf.Tensor): Input tensor representing the 3x3 rotation matrix.

    Returns:
    - np.array: Quaternion representation.
    """

    r11 = mat[0][0]
    r12 = mat[0][1]
    r13 = mat[0][2]
    r21 = mat[1][0]
    r22 = mat[1][1]
    r23 = mat[1][2] 
    r31 = mat[2][0]
    r32 = mat[2][1]
    r33 = mat[2][2] 
    
    sqrt_arg = np.array([1 + r11 + r22 + r33,
                        1 + r11 - r22 - r33,
                        1 - r11 + r22 - r33,
                        1 - r11 - r22 + r33])
    idx_max = sqrt_arg.argmax()
    arg = np.max(sqrt_arg)
    
    if idx_max == 0:
        q4 = 0.5*math.sqrt(arg)
        q1 = 1/(4*q4)*(r23 - r32)
        q2 = 1/(4*q4)*(r31 - r13)
        q3 = 1/(4*q4)*(r12 - r21)
    elif idx_max == 1:
        q1 = 0.5*math.sqrt(arg)
        q2 = 1/(4*q1)*(r12 + r21)
        q3 = 1/(4*q1)*(r13 + r31)
        q4 = 1/(4*q1)*(r23 - r32)
    elif idx_max == 2:
        q2 = 0.5*math.sqrt(arg)
        q1 = 1/(4*q2)*(r21 + r12)
        q3 = 1/(4*q2)*(r23 + r32)
        q4 = 1/(4*q2)*(r31 - r13)
    elif idx_max == 3:
        q3 = 0.5*math.sqrt(arg)
        q1 = 1/(4*q3)*(r31 + r13)
        q2 = 1/(4*q3)*(r32 + r23)
        q4 = 1/(4*q3)*(r12 - r21)   
    # Normalize and return the quaternion
    norm = math.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
    su2 = np.array([q1/norm, q2/norm, q3/norm, q4/norm])
    return su2

# Quaternion to rotation matrix domain
def su2_to_so3(su2_):
    """
    Convert a quaternion back to a 3x3 rotation matrix.
    
    Args:
    - su2_ (tf.Tensor): Input tensor representing the quaternion.

    Returns:
    - tf.Tensor: 3x3 rotation matrix.
    """
    # Extract quaternion components
    e1, e2, e3, q1 = tf.unstack(su2_, axis=-1)
    
    t2 = e1*e2*2.0
    t3 = e3*q1*2.0
    t4 = e1*e1
    t5 = e2*e2
    t6 = e3*e3
    t7 = q1*q1
    t8 = e1*e3*2.0
    t9 = e2*e3*2.0
    t10 = e1*q1*2.0
    out = tf.stack([t4-t5-t6+t7,  t2-t3,        t8+e2*q1*2.0,    
                    t2+t3,        -t4+t5-t6+t7, t9-t10,  
                    t8-e2*q1*2.0, t9+t10,       -t4-t5+t6+t7], axis=1)
    
    out = K.reshape(out,(-1, 3, 3))
    out = tf.transpose(out, perm=[0, 2, 1])
    return out
# ---------------------------------------------------------------------------------------------------------------------#
#                                                     CUSTOM LAYER                                                     #
# ---------------------------------------------------------------------------------------------------------------------#
'''''
    custom layer for trainables loss function parameters
    executed in main.py

'''''
# Define a custom Keras Layer with trainable parameters.
class TrainableSPSQLayer(Layer):

    # Constructor for the layer. We just call the parent constructor here.
    def __init__(self, **kwargs):
        super(TrainableSPSQLayer, self).__init__(**kwargs)

    # This method initializes the weights of the layer.
    def build(self, input_shape):
        
        # Add a trainable weight called 'SP' initialized with 0.0.
        self.SP = self.add_weight(name='SP',
                                  shape=(),    # Scalar weight (no shape)
                                  initializer=tf.keras.initializers.Constant(0.0), 
                                  trainable=True) # Weight is trainable
        
        # Add another trainable weight called 'SQ' initialized with -3.0.
        self.SQ = self.add_weight(name='SQ',
                                  shape=(),    # Scalar weight (no shape)
                                  initializer=tf.keras.initializers.Constant(-3.0), 
                                  trainable=True) # Weight is trainable

        # Call the parent class's build method. This ensures that the layer's build method has been called.
        super(TrainableSPSQLayer, self).build(input_shape)

    # This method describes the layer's logic. Here, it's just returning the input 'x' without any modification.
    def call(self, x):
        return x

# ---------------------------------------------------------------------------------------------------------------------#
#                                                     LOSSS                                                            #
# ---------------------------------------------------------------------------------------------------------------------#
'''''
    loss function class 
    executed in main.py
    l1 and l2 loss from paper referenced in section homoscedatic loss

    
'''''

class PoseLosses:
    
    @staticmethod
    def posLoss(y_pred, y_true):
        diff = y_pred - y_true
        diff_square = K.square(diff)
        loss = K.sum(diff_square, 1)
        loss = K.sqrt(loss)
        return loss

    @staticmethod
    def sixDimToRotMat(sixDim):
        sixDim = tf.cast(sixDim, dtype=tf.float32)
        transArr = K.reshape(sixDim, (-1, 2, 3))
        transArr = tf.transpose(transArr, perm=[0, 2, 1])
        x_raw = transArr[:,:,0]
        y_raw = transArr[:,:,1]
        x = K.l2_normalize(x_raw, axis=1)
        z = tf.linalg.cross(x, y_raw)
        z = K.l2_normalize(z, axis=1)
        y = tf.linalg.cross(z, x)
        rotMat = tf.stack([x, y, z], axis=1)
        rotMat = tf.transpose(rotMat, perm=[0, 2, 1])
        return rotMat

    @staticmethod # position loss
    def normFrobSq(tensor):
        tensorT = tf.transpose(tensor, perm=[0,2,1], conjugate=True)
        temp = tensor * tensorT
        tempDiag = tf.linalg.diag_part(temp)
        out = K.sum(tempDiag, axis=1)
        return out

    @staticmethod #6 dimensional vector loss
    def attLoss(y_pred, y_true):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        rotmatPred = PoseLosses.sixDimToRotMat(y_pred)
        rotmatTrue = PoseLosses.sixDimToRotMat(y_true)
        loss = K.sqrt(PoseLosses.normFrobSq(rotmatPred - rotmatTrue))
        return loss

    @staticmethod # translation loss
    def l1_loss(input, target, reduce='mean'):
        loss = tf.reduce_sum(tf.abs(target - input), axis=1)
        if reduce == 'none':
            return loss
        elif reduce == 'mean':
            return tf.reduce_mean(loss)
        else:
            raise Exception(f'Reduction method {reduce} not known')

    @staticmethod #6 dimensional vector loss
    def l2_loss(input, target, reduce='mean'):
        loss = tf.sqrt(tf.reduce_sum(tf.square(target - input), axis=1))
        if reduce == 'none':
            return loss
        elif reduce == 'mean':
            return tf.reduce_mean(loss)
        else:
            raise Exception(f'Reduction method {reduce} not known')

    @staticmethod # adaptive loss function SP - position weights, SQ - attitude weight
    def Adaptive_loss(SP, SQ):
        def lossTotal(y_true, y_pred):
            posPred = y_pred[:, :3]
            attPred = y_pred[:, 3:]
            posTrue = y_true[:, :3]
            attTrue = y_true[:, 3:]
            loss = PoseLosses.l1_loss(posPred, posTrue) * K.exp(-SP) + SP + PoseLosses.l2_loss(attPred, attTrue) * K.exp(-SQ) + SQ
            return loss
        return lossTotal
    

# ---------------------------------------------------------------------------------------------------------------------#
#                                                     HOMOSCEDATIC LOSSS                                               #
# ---------------------------------------------------------------------------------------------------------------------#
     
    '''''  
    The loss function is for eager execution 
    Homoscedatic loss function borrowed from paper cited below from git repository, 
    the code converted from Pytorch environment to tensorflow/keras
    
    see reference below
    '''''


    '''''
        @article{boittiaux2022homographyloss,
    author={Boittiaux, Cl\'ementin and
            Marxer, Ricard and
            Dune, Claire and
            Arnaubec, Aur\'elien and
            Hugel, Vincent},
    journal={IEEE Robotics and Automation Letters},
    title={Homography-Based Loss Function for Camera Pose Regression},
    year={2022},
    volume={7},
    number={3},
    pages={6242-6249},
    }
    '''''
def l1_loss(input, target, reduce='mean'):
    loss = tf.reduce_sum(tf.abs(target - input), axis=1)
    if reduce == 'none':
        return loss
    elif reduce == 'mean':
        return tf.reduce_mean(loss)
    else:
        raise Exception(f'Reduction method {reduce} not known')

def l2_loss(input, target, reduce='mean'):
    loss = tf.sqrt(tf.reduce_sum(tf.square(target - input), axis=1))
    if reduce == 'none':
        return loss
    elif reduce == 'mean':
        return tf.reduce_mean(loss)
    else:
        raise Exception(f'Reduction method {reduce} not known')

class HomoscedasticLoss(tf.keras.layers.Layer):
    def __init__(self, s_hat_t, s_hat_q, device='CPU:0'):
        super(HomoscedasticLoss, self).__init__()
        with tf.device(device):
            self.s_hat_t = tf.Variable(initial_value=tf.constant(s_hat_t, dtype=tf.float32), trainable=True)
            self.s_hat_q = tf.Variable(initial_value=tf.constant(s_hat_q, dtype=tf.float32), trainable=True)

    def call(self, y_true, y_pred, training=None):
        LtI = l1_loss(y_true[:, :3], y_pred[:, :3])  # Using first three values
        LqI = l2_loss(y_true[:, 3:], y_pred[:, 3:])  # Using the remaining six values
        
        error = LtI * tf.exp(-self.s_hat_t) + self.s_hat_t + LqI * tf.exp(-self.s_hat_q) + self.s_hat_q

        self.add_loss(error)
        return error



    # ---------------------------------------------------------------------------------------------------------------------#
    #                                                     CALLBACKS                                                                #
    # ---------------------------------------------------------------------------------------------------------------------#

class PrintSPSQ(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to print the values of SP and SQ at the end of each epoch.

    This callback is designed for a model that contains a layer named 'trainable_spsq_layer',
    which has two trainable parameters, SP and SQ.

    Attributes:
    - None

    Methods:
    - on_epoch_end: Overridden method from the base Callback class. 
                    Called at the end of each epoch, and prints the current values of SP and SQ.
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        This method is automatically called by Keras at the end of each epoch during training.

        Args:
        - epoch: The current epoch number (0-based).
        - logs: A dictionary of current training metrics. Not used in this method.

        Returns:
        - None. This method prints the values of SP and SQ.
        """
        
        # Get the SP value from the 'custom_trainable_params_layer' of the model.
        sp_value = self.model.get_layer('trainable_spsq_layer').SP.numpy()

        # Get the SQ value from the 'custom_trainable_params_layer' of the model.
        sq_value = self.model.get_layer('trainable_spsq_layer').SQ.numpy()

        # Print the values of SP and SQ.
        print(f"\nEnd of epoch {epoch}. SP: {sp_value}, SQ: {sq_value}")

    # ---------------------------------------------------------------------------------------------------------------------#
    #                                                     MISCELANIOUS                                                     #
    # ---------------------------------------------------------------------------------------------------------------------#


    
class IMGSorter:
    def __init__(self, base_path):
        """
        Initializes the IMGSorter class with the base directory path.

        Args:
        - base_path (str): Directory path containing the images to be sorted.
        """
        self.base_path = base_path

    def get_number_before_zero(self, image_path):
        """
        Extracts the three numbers before the zero in the image filename.

        Args:
        - image_path (str): Full path to the image.

        Returns:
        - Tuple of three integers extracted from the filename.
        """
        filename = os.path.basename(image_path)
        parts = filename.split('.')
        parts_ = parts[0].split('_')
        x = int(parts_[1])
        z = int(parts_[2])
        y = int(parts_[3])
        return x, z, y

    def sort_images_by_number_before_zero(self, image_paths):
        """
        Sorts a list of image paths based on the numbers extracted from their filenames.

        Args:
        - image_paths (list): List of image paths to be sorted.

        Returns:
        - Sorted list of image paths.
        """
        return sorted(image_paths, key=self.get_number_before_zero)

    def get_sorted_image_paths(self, format):
        """
        Retrieves image paths from the base directory and sorts them based on their filenames.

        Args:
        - format (str): File format/extension of the images to be sorted (e.g., '.exr', '.png', '.txt').

        Returns:
        - Sorted list of image paths.
        """
        image_paths = [os.path.join(self.base_path, filename) for filename in os.listdir(self.base_path) if filename.endswith(str(format))]
        return self.sort_images_by_number_before_zero(image_paths)
    
    def get_image_path(self, filename):
        """
        Constructs the full path for a given image filename.

        Args:
        - filename (str): Name of the image file.

        Returns:
        - Full path to the image file.
        """
        image_path = os.path.join(self.base_path, filename)
        return image_path

