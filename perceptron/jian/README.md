## TODO
- fully connected neural network for Chinese character classification

## Dataset
- `kagglehub`: `chinese-mnist`

The `datasets.py` is a module for autmated dataset preprocessing and splitting. The following Important varibles will be generated when using:

```python
from datasets import train_dataset, val_dataset, test_dataset, device
```

- `train_data` : `val_dataset` : `test_dataset` = 0.7 : 0.15 : 0.15

## FNN
The `FNN.py` module defines the fully connected neural network (or perceptron).

## Hyperparameter search
The `hyperparameters_search.py` performs hyperparameter search for the FNN. The current demo only searches the following
- neurons in the first hidden layer
- neurons in the second hidden layer
- dropout rate
- activation function

```bash
python hyperparameters_search.py
```
This will generate a result csv sheet in the `./results/` directory.

## Training
After finding out the best combination of hyperparameters, we train the model again with that hyperparamter combination with more epochs until reaching convergence.

```bash
python train.py
```

The current model configuration, weights and training/validation loss and accuracy history will be stored as data files in the `./results` directory.

## Testing
Finally, we want to evaluate the trained model on the splited testing dataset which has never been seen by the model.

```bash
python test.py ./results/model_config*.json ./results/weights_*.pt
```

The json file and pt file are generated during the training stage. Runing this command will directly plot the confusion matrix and print out the testing accuracy.
