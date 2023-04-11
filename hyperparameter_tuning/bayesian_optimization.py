import numpy as np
from functools import partial
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import TimeSeriesSplit
from models import *

def optimize_lstm(data):
    # Define the search space for LSTM hyperparameters
    search_space = {
        'num_layers': Integer(1, 3),
        'hidden_size': Integer(32, 128),
        'dropout': Real(0.0, 0.5),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    }
    # ... (Add other hyperparameters as needed)

    model = LSTMModel()  # Replace with your LSTM model class
    return bayesian_optimize(model, data, search_space)


def optimize_gru(data):
    # Define the search space for GRU hyperparameters
    search_space = {
        'num_layers': Integer(1, 3),
        'hidden_size': Integer(32, 128),
        'dropout': Real(0.0, 0.5),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    }
    # ... (Add other hyperparameters as needed)

    model = GRUModel()  # Replace with your GRU model class
    return bayesian_optimize(model, data, search_space)


def optimize_cnn(data):
    # Define the search space for CNN hyperparameters
    search_space = {
        'num_layers': Integer(1, 5),
        'num_filters': Integer(16, 128),
        'kernel_size': Integer(2, 5),
        'dropout': Real(0.0, 0.5),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    }
    # ... (Add other hyperparameters as needed)

    model = CNNModel()  # Replace with your CNN model class
    return bayesian_optimize(model, data, search_space)


def optimize_transformer(data):
    # Define the search space for Transformer hyperparameters
    search_space = {
        'num_layers': Integer(1, 6),
        'num_heads': Integer(1, 8),
        'hidden_dim': Integer(32, 128),
        'dropout': Real(0.0, 0.5),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    }
    # ... (Add other hyperparameters as needed)

    model = TransformerModel()  # Replace with your Transformer model class
    return bayesian_optimize(model, data, search_space)


def bayesian_optimize(model, data, search_space):
    optimizer = BayesSearchCV(
        model,
        search_space,
        n_iter=50,
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1,
        scoring='neg_mean_squared_error',
    )
    optimizer.fit(data[:, :-1], data[:, -1])
    best_params = optimizer.best_params_
    best_score = -optimizer.best_score_

    return best_params, best_score


def optimize(model_type, data):
    if model_type == 'LSTM':
        return optimize_lstm(data)
    elif model_type == 'GRU':
        return optimize_gru(data)
    elif model_type == 'CNN':
        return optimize_cnn(data)
    elif model_type == 'Transformer':
        return optimize_transformer(data)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

