import tensorflow as tf
import keras.backend as K

class AdaptiveLoss(tf.keras.losses.Loss):
    def __init__(self, SP_initial, SQ_initial):
        super(AdaptiveLoss, self).__init__()
        self.SP = tf.Variable(SP_initial)
        self.SQ = tf.Variable(SQ_initial)
        self.last_pos_loss = None  # Attributes to store the last computed losses
        self.last_att_loss = None

    def call(self, y_true, y_pred):
        posPred = y_pred[:,:3]
        attPred = y_pred[:,3:]
        posTrue = y_true[:,:3]
        attTrue = y_true[:,3:]

        pos_loss = self.posLoss(posPred, posTrue) * tf.exp(-self.SP) + self.SP
        att_loss = self.attLoss(attPred, attTrue) * tf.exp(-self.SQ) + self.SQ

        self.last_pos_loss = K.mean(pos_loss)
        self.last_att_loss = K.mean(att_loss)

        total_loss = pos_loss + att_loss
        return total_loss
    
    def get_last_pos_loss(self):
        return self.last_pos_loss

    def get_last_att_loss(self):
        return self.last_att_loss

    def posLoss(self, y_pred, y_true):
        diff = y_pred - y_true
        diff_squre = K.square(diff)
        loss = K.sum(diff_squre, axis=1)
        loss = K.sqrt(loss)
        return loss

    def sixDimToRotMat(self, sixDim):
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

    def normFrobSq(self, tensor):
        tensorT = tf.transpose(tensor, perm=[0, 2, 1], conjugate=True)
        temp = tensor * tensorT
        tempDiag = tf.linalg.diag_part(temp)
        out = K.sum(tempDiag, axis=1)
        return out

    def attLoss(self, y_pred, y_true):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
   
        rotmatPred = self.sixDimToRotMat(y_pred)
        rotmatTrue = self.sixDimToRotMat(y_true)
        loss = K.sqrt(self.normFrobSq(rotmatPred - rotmatTrue))
        return loss

# # Example usage
# SP_initial = 1.0
# SQ_initial = 1.0
# adaptive_loss = AdaptiveLoss(SP_initial, SQ_initial)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# # Training loop
# for epoch in range(num_epochs):
#     for batch_inputs, batch_targets in dataset:
#         with tf.GradientTape() as tape:
#             predictions = model(batch_inputs)
#             loss = adaptive_loss(batch_targets, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables + [adaptive_loss.SP, adaptive_loss.SQ])
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables + [adaptive_loss.SP, adaptive_loss.SQ]))
