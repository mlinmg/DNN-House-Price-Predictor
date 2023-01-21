import pandas as pd
from keras.saving.save import load_model
from keras.utils import losses_utils, plot_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
import tensorflow as tf
from keras import layers
import seaborn as sns
from keras import regularizers

# DNN Variables
num_epochs = 100
batch_size = 32
l1_reg = 0.1
l2_reg = 0.01
num_neurons1 = 512#len(data.columns)
num_neurons2 = 512
num_neurons3 = 256
num_neurons4 = 128
num_neurons5 = 1

#Splitting dataset
X = data.drop(columns=["target", "Indirizzo", "Unnamed: 0"])
y = data["target"]


# Creating aclass for converting string into numeric values
class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

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
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output


multi = MultiColumnLabelEncoder(
    columns=['Contratto', 'Tipologia', 'Piano', 'Stato', 'Climatizzazione', 'City', "Riscaldamento"])
X = multi.fit_transform(X)

# Divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Dataset creation
try:
    model = load_model("deep_learning_model.h5")
except:

    data = pd.read_csv("dataset.csv")

    # Model creation
    model = keras.Sequential()

    # Aggiunta of layer con regularization and dropout
    model.add(layers.Dense(num_neurons1, input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_neurons2, kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_neurons3, kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_neurons4, kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))


    # Aggiunta del output layer
    model.add(layers.Dense(num_neurons5))
    model.add(layers.Activation('linear'))

    # Compilazione del modello utilizzando l'ottimizzatore Adam
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # Addestramento del modello
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Valutazione del modello sui dati di test
test_loss = model.evaluate(X_test, y_test)
# Stampa del valore della mean absolute error sul test set
print("Mean Absolute Error on test set:", test_loss)

# Previsione del prezzo utilizzando il modello addestrato
y_pred = model.predict(X_test)
# Stampa del valore della mean absolute error tra le previsioni e i valori reali
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Recupero dei valori di perdita e accuratezza per ogni epoca
loss = history.history['loss']
val_loss = history.history['val_loss']
loss = pd.DataFrame(history.history['loss'], columns=['loss'])
val_loss = pd.DataFrame(history.history['val_loss'], columns=['val_loss'])
loss_rolling = loss['loss'].rolling(window=10).mean()
val_loss_rolling = val_loss['val_loss'].rolling(window=10).mean()

# Creazione del grafico
plt.figure()
plt.plot(loss.index, loss_rolling, label="Perdita Allenamento")
plt.plot(val_loss.index, val_loss_rolling, label="Perdita Validazione")
plt.xlabel("Epoche")
plt.ylabel("Perdita")
plt.title("Epoche: {}, Batch: {}, L1: {}, L2: {}, Neuroni: input: {}, hidden#1: {}, hidden#2: {}, hidden#3: {}, output: {}".format(num_epochs, batch_size, l1_reg, l2_reg, num_neurons1, num_neurons2, num_neurons3, num_neurons4, num_neurons5))
plt.legend()

# Salva il modello
model.save("deep_learning_model.h5")

# Grafico della curva di apprendimento
plt.show()