# ==================model.base_loss.py=====================
# This module implements some defined losses which are not
# in the Keras.

# Version: 1.0.0
# Date: 2019.08.07
# ============================================================

from keras import backend as k
import tensorflow as tf


###############################################################
# Focal Loss
###############################################################
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return - k.mean(alpha * k.pow(1. - pt_1, gamma) * k.log(pt_1)) \
               - k.mean((1 - alpha) * k.pow(pt_0, gamma) * k.log(1. - pt_0))
    return focal_loss_fixed


