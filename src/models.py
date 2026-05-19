"""
This module houses all model definitions, any configurations or hyperparameters can be passed in as arguments during class construction.

List of architectures to implement:
1. Baseline CNN (✔️)
2. VGG16
3. ResNet34 (✔️)
4. EfficientNet (✔️)
"""

import tensorflow as tf

class BaselineBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.stack = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(3,3),
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(
                pool_size=(2,2), # Strides defaults to pool size (2)
                padding="valid",
            ),
        ])

    def call(self, inputs, training=False):
        return self.stack(inputs, training=training)
        

class BaselineCNN(tf.keras.Model):
    """
    This is a implementation of a naive baseline CNN comprised of:

    Downsampling and feature extraction:
    1. Conv2D (32 filters, 3x3 kernel, stride 1, ReLU)
    2. MaxPool (2x2 kernel)
    3. Conv2D (64 filters, 3x3 kernel, ReLU)
    4. MaxPool (2x2 kernel)
    
    Classification head:
    5. Flatten
    6. Dense 128, ReLU
    7. Dropout 0.2
    8. Dense 1, Sigmoid

    We'll omit batch normalisation from this architecture.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.stack = tf.keras.Sequential([
            BaselineBlock(filters=32), # 112 x 112, 32 feature maps
            BaselineBlock(filters=64), # 56 x 56, 64 feature maps
        ])
            
        # Classification head (flatten, linear projection, dropout, final sigmoid unit)
        self.stack.add(tf.keras.layers.Flatten())
        self.stack.add(tf.keras.layers.Dense(128, activation="relu"))
        self.stack.add(tf.keras.layers.Dropout(0.2))
        self.stack.add(tf.keras.layers.Dense(1, activation="sigmoid")) # Final classification, bounded between 0 - 1 (predicted proba)
        
    def call(self, inputs, training=False):
        return self.stack(inputs, training=training)
    

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

    def call(self, inputs, training=False):
        return tf.nn.relu(
            tf.add( self.stack(inputs, training=training), self.skip_connection(inputs, training=training) ) # Element wise tensor addition
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
                pool_size=(3,3),
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

    def call(self, inputs, training=False):
        return self.stack(inputs, training=training)



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