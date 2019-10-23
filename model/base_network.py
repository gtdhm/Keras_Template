# ==================model.base_network.py=====================
# This module implements a base network for the project.

# Version: 1.0.0
# Date: 2019.08.07
# ============================================================

import keras.layers as kl


###############################################################
# BaseNetwork Graph
###############################################################
def BaseNetwork(x_input, cfg):
    """This graph includes base architecture for the network.
    Inputs:
        x_input: (b, h, w, c) tensor of the kl.Input()
        cfg: the total options
    """
    # TODO(Usr) >>> redefine following layers
    # Define network layers
    """convolution stage"""
    # (224, 224, 3)
    x = kl.Conv2D(16, (5, 5), strides=1, padding="valid", name="conv1")(x_input)
    x = kl.Activation('relu', name='relu1')(x)
    c1 = kl.MaxPooling2D((2, 2), name='max_pool1')(x)
    # (110, 110, 16)
    x = kl.Conv2D(32, (3, 3), strides=1, padding="valid", name="conv2")(c1)
    x = kl.Activation('relu', name='relu2')(x)
    c2 = kl.MaxPooling2D((2, 2), name='max_pool2')(x)
    # (54, 54, 32)
    x = kl.Conv2D(64, (3, 3), strides=1, padding="valid", name="conv3")(c2)
    x = BatchNorm(name='bn1')(x, training=cfg.opts.train_bn)
    x = kl.Activation('relu', name='relu3')(x)
    c3 = kl.MaxPooling2D((2, 2), name='max_pool3')(x)
    # (26, 26, 64)
    x = kl.Conv2D(128, (3, 3), strides=1, padding="valid", name="conv4")(c3)
    x = BatchNorm(name='bn2')(x, training=cfg.opts.train_bn)
    x = kl.Activation('relu', name='relu4')(x)
    c4 = kl.MaxPooling2D((2, 2), name='max_pool4')(x)
    # (12, 12, 128)
    x = kl.Conv2D(256, (1, 1), strides=1, padding="same", name="conv5")(c4)
    x = BatchNorm(name='bn3')(x, training=cfg.opts.train_bn)
    x = kl.Activation('relu', name='relu5')(x)
    c5 = kl.MaxPooling2D((2, 2), name='max_pool5')(x)
    # (6, 6, 256)

    """fully connected stage"""
    # convert it to a vector
    x = kl.Flatten(name="flatten")(c5)
    x = kl.Dense(128, activation='relu', name="fc1")(x)
    out = kl.Dense(len(cfg.class_name), activation='softmax', name='fc2')(x)

    # TODO(User): End
    return [c1, c2, c3, c4, c5, out]


############################################################
#  BatchNormalization layer
############################################################
class BatchNorm(kl.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)




