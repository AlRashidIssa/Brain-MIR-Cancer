import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, # type: ignore
                                     LeakyReLU, MaxPooling2D, Dropout, 
                                     UpSampling2D, Concatenate, GlobalAveragePooling2D,
                                     Dense, Flatten)
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


class ClassificationModel:

    def __init__(self, input_shape: tuple = (256, 256, 3)) -> None:
        """Initialize the ClassificationModel with an input shape."""
        self.inputs = Input(shape=input_shape)

    def classification(self) -> Model:
        """Builds the classification model.
        
        Returns:
            Model: A Keras Model object representing the classification model.
        """
        # Initial Convolutional Block to adapt the input
        x = Conv2D(3, (1, 1))(self.inputs)

        # Encoder: Series of Convolutional and Pooling layers
        x1 = self.conv_block(x, 64)
        x2 = self.conv_block(x1, 64)
        x3 = self.conv_block(x2, 64)
        x4 = self.conv_block(x3, 64)
        x5 = self.conv_block(x4, 64)

        # Deepest Convolutional Layer
        x6 = self.conv_block(x5, 64)

        # Decoder: Series of Upsampling and Convolutional layers
        conv = self.upsample_and_concat(x6, x5, 512)
        conv = self.upsample_and_concat(conv, x4, 256)
        conv = self.upsample_and_concat(conv, x3, 128)
        conv = self.upsample_and_concat(conv, x2, 64)
        conv = self.upsample_and_concat(conv, x1, 32)

        # Final Convolutional Layer
        conv = Conv2D(16, (1, 1), padding='same')(conv)
        conv = Conv2D(8, (1, 1), padding='same')(conv)
        conv = Conv2D(4, (1, 1), padding='same')(conv)
        conv = Conv2D(3, (1, 1), padding='same')(conv)

        # Concatenate with original input
        c6 = Concatenate()([conv, x])
        conv = Conv2D(3, (1, 1), padding='same')(c6)

        # Additional Convolutional Layers
        x = self.conv_block(conv, 64)
        x = self.conv_block(x, 64)
        x = self.conv_block(x, 64)
        x = self.conv_block(x, 64)

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        
        # Flatten before Fully Connected Layers
        x = Flatten()(x)

        # Fully Connected Layers (FCL)
        x = self.fully_connected_block(x, 4096)
        x = self.fully_connected_block(x, 4096)
        x = self.fully_connected_block(x, 4096)
        x = self.fully_connected_block(x, 4096)

        
        # Output Layer
        outputs = Dense(1, activation='sigmoid')(x)

        # Create model
        model = Model(inputs=self.inputs, outputs=outputs)
        return model

    def conv_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """Applies two convolutional layers with Batch Normalization and Leaky ReLU, followed by MaxPooling."""
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    def upsample_and_concat(self, x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
        """Upsamples the input and concatenates it with a skip connection from the encoder."""
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filters, (1, 1), padding='same')(x)
        x = Concatenate()([x, skip])
        x = Conv2D(filters, (1, 1), padding='same')(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        return x

    def fully_connected_block(self, x: tf.Tensor, units: int) -> tf.Tensor:
        """Applies a series of fully connected layers with L2 regularization."""
        x = Dense(units, activation='relu', kernel_regularizer=l2(1e-4))(x)
        return x
