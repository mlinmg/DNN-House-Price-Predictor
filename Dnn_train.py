import pandas as pd
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from keras import layers
from keras import regularizers

# Dataset creation
data = pd.read_csv("dataset.csv")
# Splitting dataset
X = data.drop(columns=["target", "Indirizzo", "Unnamed: 0"])
y = data["target"]


# Creating a class for converting string into numeric values
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


# Creating a function for the addition of a layer
def input_layer_addition(neurons, regularizer, activation, dropout, input_shape="(X_train.shape[1])"):
    model.add(layers.Dense(neurons, input_shape=input_shape, kernel_regularizer=regularizers.l1(regularizer)))
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(dropout))


def hidden_layer_addition(neurons, regularizer, activation, dropout):
    model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l1(regularizer)))
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(dropout))


def output_layer_addition(neurons, activation):
    model.add(layers.Dense(neurons))
    model.add(layers.Activation(activation))


def plot_maker(historyf, epochs, batch, l1, neurons1, neurons2, neurons3, neurons4,
               neurons5):
    # Getting loss and validation loss from the model history
    loss = pd.DataFrame(historyf.history['loss'], columns=['loss'])
    val_loss = pd.DataFrame(historyf.history['val_loss'], columns=['val_loss'])
    # Making the mean of such values
    loss_rolling = loss['loss'].rolling(window=10).mean()
    val_loss_rolling = val_loss['val_loss'].rolling(window=10).mean()
    plt.figure()
    plt.plot(loss.index, loss_rolling, label="Training Loss")
    plt.plot(val_loss.index, val_loss_rolling, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Epoche: {}, Batch: {}, L1: {}, Neuroni: input: {}, hidden#1: {}, hidden#2: {}, hidden#3: {}, output: {}".format(
            epochs, batch, l1, neurons1, neurons2, neurons3, neurons4,
            neurons5))
    plt.legend()
    # Show the graph
    plt.show()


# DNN Variables
num_epochs = 100
batch_size = 32
l1_reg = 0.1
num_neurons1 = 512
num_neurons2 = 512
num_neurons3 = 256
num_neurons4 = 128
num_neurons5 = 1

# Creating a multilabel object and using the function fit_transform
multi = MultiColumnLabelEncoder(
    columns=['Contratto', 'Tipologia', 'Piano', 'Stato', 'Climatizzazione', 'City', "Riscaldamento"])
X = multi.fit_transform(X)

# Divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Try to load a previous created model
try:
    model = load_model("deep_learning_model.h5")
except:
    # Model creation
    model = keras.Sequential()
    # Layers addition
    input_layer_addition(num_neurons1, l1_reg, 'relu', 0.5)
    hidden_layer_addition(num_neurons2, l1_reg, 'relu', 0.5)
    hidden_layer_addition(num_neurons3, l1_reg, 'relu', 0.5)
    hidden_layer_addition(num_neurons4, l1_reg, 'relu', 0.5)
    output_layer_addition(num_neurons5, 'linear')
    # Model compiling using the adam optimizer
    model.compile(loss='mean_absolute_error', optimizer='adam')

# Model training
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Valutazione del modello sui dati di test
test_loss = model.evaluate(X_test, y_test)
# Stampa del valore della mean absolute error sul test set
print("Mean Absolute Error on test set:", test_loss)

# Previsione del prezzo utilizzando il modello addestrato
y_pred = model.predict(X_test)
# Stampa del valore della mean absolute error tra le previsioni e i valori reali
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Creazione del grafico
plot_maker(history, num_epochs, batch_size, l1_reg, num_neurons1, num_neurons2, num_neurons3, num_neurons4,
           num_neurons5)

# Salva il modello
model.save("deep_learning_model.h5")
