# Italian House Price Predictor - DNN

This is a project in which I built and trained an AI, capable of predicting the house prices in Italy.
## Installation

Copy this repository on your machine.

```bash
git clone https://github.com/mlinmg/AI_GPU_try.git
```

## Usage
You can choose to either use the pre-trained model(deep_learning_model.h5) or to train one with your own. If you choose to do so, keep in mind that you have to follow the dasatet structure. with those command:
```python
from keras.saving.save import load_model

# insert your data here
data = yourdata

# load the model
model = load_model('model.h5')

# make the prediction 
y_pred = model.predict(insert)

# print the error and the result
print("The cost of the house is:" + str(y_pred))
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)