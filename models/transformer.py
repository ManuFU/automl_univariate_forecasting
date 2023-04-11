import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    # Generate the positional encoding matrix
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(0, d_model, 2), d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    # Compute the angle for positional encoding
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * i) / np.float32(d_model))
        return position * angles

    # Add the positional encoding to the input embeddings
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(d_model)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    # Define the forward pass for the Transformer block
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(Model):
    def __init__(self, hyperparameters=None):
        super(TransformerModel, self).__init__()

        if hyperparameters is None:
            self.num_layers = 2
            self.d_model = 64
            self.num_heads = 8
            self.ff_dim = 128
            self.input_shape = 20
            self.dropout_rate = 0.1
            self.learning_rate = 0.001
        else:
            self.num_layers = hyperparameters['num_layers']
            self.d_model = hyperparameters['d_model']
            self.num_heads = hyperparameters['num_heads']
            self.ff_dim = hyperparameters['ff_dim']
            self.input_shape = hyperparameters['input_shape']
            self.dropout_rate = hyperparameters['dropout_rate']
            self.learning_rate = hyperparameters['learning_rate']

        self.build_model()

    def build_model(self):
        self.embedding = Dense(self.d_model)
        self.pos_encoding = PositionalEncoding(self.input_shape, self.d_model)
        self.transformer_blocks = [TransformerBlock(self.d_model, self.num_heads, self.ff_dim, self.dropout_rate) for _
                                   in range(self.num_layers)]
        self.dense = Dense(1)
        self.optimizer = Adam(learning_rate=self.learning_rate)
        super().compile(optimizer=self.optimizer, loss='mse')

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, training)
        return self.dense(x)

    def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.history = super().fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                   validation_split=validation_split, verbose=verbose)

    def predict(self, X_test):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return super().predict(X_test)
