import numpy as np
from keras import regularizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import keras

data = pd.read_csv("dataset.csv")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
X = data.drop(columns=["target", "Indirizzo"])
y = data["target"]

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

multi = MultiColumnLabelEncoder(columns=['Contratto', 'Tipologia', 'Piano', 'Stato', 'Climatizzazione', 'City',"Riscaldamento"])
X = multi.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(trial):
    n_inp = trial.suggest_int('n_inp', 128, 512)
    n_hidden = trial.suggest_int('n_hidden', 1, 4)
    units = trial.suggest_int('units', 64, 512)
    l1_reg = trial.suggest_float('l1', 0, 2)
    dropout = 0.5
    model = keras.Sequential()
    # Aggiunta della prima hidden layer con L1 regularization
    model.add(keras.layers.Dense(n_inp, input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    for i in range(n_hidden):
        model.add(keras.layers.Dense(units=units, activation='relu', kernel_regularizer=regularizers.l1(l1_reg)))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def objective(trial):
    batch = 32
    model = create_model(trial)
    history = model.fit(X_train, y_train, batch_size=batch, epochs=100, validation_data=(X_test, y_test), verbose=0)
    val_acc = min(history.history['mean_squared_error'])
    return val_acc


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

print("Best hyperparameters found:", study.best_params)
