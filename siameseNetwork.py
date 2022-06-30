from keras import layers
from keras import Model
import keras.backend as K
import tensorflow as tf


class SiameseNetwork:
    def __init__(self, width, height, channel, target_shape):
        self.width = width
        self.height = height
        self.channel = channel
        self.siamese_model = self.build_model(target_shape)

    def network(self , name):
        inputs = layers.Input((self.width, self.height, self.channel))

        x = layers.Conv2D(64, (10, 10), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(128, (7, 7), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(128, (4, 4), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(256, (4, 4), padding="same", activation="relu")(x)
        fcOutput = layers.Flatten()(x)
        fcOutput = layers.Dense(4096, activation="relu")(fcOutput)
        outputs = layers.Dense(1024, activation="sigmoid")(fcOutput)

        embedding = Model(inputs, outputs, name=name)
        return embedding

    def build_model(self, target_shape):
        anchor_input = layers.Input(name="anchor", shape=target_shape)
        compare_input = layers.Input(name="compare", shape=target_shape)

        distances = DistanceLayer()(
            self.network("Embedding1")(anchor_input),
            self.network("Embedding2")(compare_input),
        )

        outputs = layers.Dense(1, activation="sigmoid")(distances)

        siamese_model = Model(
            inputs=[anchor_input, compare_input], outputs=outputs
        )
        siamese_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return siamese_model

    def contrastive_loss(y, preds, margin=1):
        y = tf.cast(y, preds.dtype)
        squaredPreds = K.square(preds)
        squaredMargin = K.square(K.maximum(margin - preds, 0))
        loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

        return loss

    def save(self, name):
        self.siamese_model.save_weights(name)

    def load(self, name):
        self.siamese_model.load_weights(name)


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance
    between the embeddings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, compare):
        sum_squared = K.sum(K.square(anchor - compare), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))
