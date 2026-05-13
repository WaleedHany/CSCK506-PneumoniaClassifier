"""
This module houses all model definitions, any configurations or hyperparameters can be passed in as arguments during class construction.

Potential list of architectures to implement:
LeNet-5 (will very likely underperform, but worth showing for the comparison)
VGG16
ResNet34 (✔️)
EfficientNet (✔️)
"""

import tensorflow as tf
from functools import partial

class ResUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)

        # Main stack of conv layers in the residual unit
        self.stack = tf.keras.Sequential([
            tf.keras.layers.Conv2D( # TF infers the number of input channels lazily (1)
                filters=filters, # Feature maps / output channels
                kernel_size=(3,3),
                strides=strides,
                padding="same",
                kernel_initializer="he_normal", # Match with ReLU activation
                use_bias=False, # Batch normalisation renders biases redundant
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(3,3),
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization()
        ])

        # The skip connection layer for when number of feature maps changes
        if strides == 2: # This conditional coincides with feature map depth changing
            self.skip_connection = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(1,1),
                    strides=strides,
                    padding="same",
                    kernel_initializer="he_normal",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization()
            ])
        else: # stride == 1
            self.skip_connection = tf.keras.layers.Identity() # Essentially a no-op

    def call(self, inputs):
        return tf.nn.relu(
            tf.add( self.stack(inputs), self.skip_connection(inputs) ) # Element wise tensor addition
        )

class ResNet34(tf.keras.Model):
    """
    This is an implementation of the 34-layer variant from https://arxiv.org/pdf/1512.03385. See page 4 for diagram.

    3 RUs with 64 feature maps
    4 RUs with 128 feature maps
    6 RUs with 256 feature maps
    3 RUs with 512 feature maps
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the starting stack of layers, residual units will be appended to the stack
        self.stack = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(7,7),
                strides=2,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(
                pool_size=3,
                strides=2,
                padding="same",
            )
        ])

        # Residual unit stack
        previous_fm = 64
        fmaps = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
        for current_fm in fmaps:
            self.stack.add(
                ResUnit(
                    filters=current_fm,
                    strides=1 if current_fm == previous_fm else 2
                )
            )
            previous_fm = current_fm
            
        # Classification head (global average pooling, final sigmoid unit)
        self.stack.add(tf.keras.layers.GlobalAveragePooling2D()) # Final feature maps = 512
        self.stack.add(tf.keras.layers.Dense(1, activation="sigmoid")) # Final classification, bounded between 0 - 1 (predicted proba)

    def call(self, inputs):
        return self.stack(inputs)



class EfficientNetBinaryClassifier(tf.keras.Model):

    def __init__(
        self,
        input_shape=(224, 224, 3),
        train_backbone=False,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )

        self.backbone.trainable = train_backbone

        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.hidden = tf.keras.layers.Dense(
            256,
            activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )

    def call(self, inputs, training=False):

        x = self.backbone(
            inputs,
            training=training
        )

        x = self.pool(x)

        x = self.dropout(
            x,
            training=training
        )

        x = self.hidden(x)

        return self.output_layer(x)