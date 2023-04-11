from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class CNNModel:
    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            # Set default hyperparameters
            self.num_layers = 2
            self.num_filters = 32
            self.kernel_size = 3
            self.dropout = 0.2
            self.learning_rate = 0.001
        else:
            self.num_layers = hyperparameters['num_layers']
            self.num_filters = hyperparameters['num_filters']
            self.kernel_size = hyperparameters['kernel_size']
            self.dropout = hyperparameters['dropout']
            self.learning_rate = hyperparameters['learning_rate']

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Add the first Conv1D layer
        model.add(Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu', input_shape=(None, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(self.dropout))

        # Add remaining Conv1D layers
        for _ in range(self.num_layers - 1):
            model.add(Conv1D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0):
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

    def predict(self, X_test):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return self.model.predict(X_test)
