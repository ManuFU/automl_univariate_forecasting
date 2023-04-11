from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class GRUModel:
    def __init__(self, hyperparameters=None):
        if hyperparameters is None:
            # Set default hyperparameters
            self.num_layers = 2
            self.hidden_size = 64
            self.dropout = 0.2
            self.learning_rate = 0.001
        else:
            self.num_layers = hyperparameters['num_layers']
            self.hidden_size = hyperparameters['hidden_size']
            self.dropout = hyperparameters['dropout']
            self.learning_rate = hyperparameters['learning_rate']

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Add the first GRU layer
        model.add(GRU(units=self.hidden_size, activation='relu', return_sequences=True, input_shape=(None, 1)))

        # Add remaining GRU layers with dropout
        for _ in range(self.num_layers - 1):
            model.add(GRU(units=self.hidden_size, activation='relu', return_sequences=True))
            model.add(Dropout(self.dropout))

        model.add(GRU(units=self.hidden_size, activation='relu'))
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
