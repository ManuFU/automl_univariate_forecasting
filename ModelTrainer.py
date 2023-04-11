import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from hyperparameter_tuning.bayesian_optimization import optimize
from models import *


class ModelTrainer:
    def __init__(self, data, model_type, hyperparameters=None):
        self.data = data
        self.model_type = model_type
        self.hyperparameters = hyperparameters

    def create_model(self, model_type, hyperparameters=None):
        if model_type == 'LSTM':
            model = LSTMModel(hyperparameters)
        elif model_type == 'GRU':
            model = GRUModel(hyperparameters)
        elif model_type == 'CNN':
            model = CNNModel(hyperparameters)
        elif model_type == 'Transformer':
            model = TransformerModel(hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model

    def train_and_evaluate(self, model, X_train, y_train, X_val, y_val):
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = mean_squared_error(y_val, predictions)
        return score

    def cross_validate(self, model, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_index, val_index in tscv.split(self.data):
            X_train, y_train = self.data[train_index, :-1], self.data[train_index, -1]
            X_val, y_val = self.data[val_index, :-1], self.data[val_index, -1]

            score = self.train_and_evaluate(model, X_train, y_train, X_val, y_val)
            scores.append(score)

        return np.mean(scores)

    def train_and_tune(self):
        if self.hyperparameters is None:
            best_hyperparameters = optimize(self.model_type, self.data)
            self.hyperparameters = best_hyperparameters

        model = self.create_model(self.model_type, self.hyperparameters)
        mean_cv_score = self.cross_validate(model)
        return model, mean_cv_score
